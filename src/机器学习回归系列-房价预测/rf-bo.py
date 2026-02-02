import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1) 数据集（回归）
def read_data(filename):
    filename = filename
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
             'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
    dataset = pd.read_csv(filename, names=names, delim_whitespace=True)
    print(dataset)
    df = pd.DataFrame(dataset)

    #  划分数据集
    features = names[:-1]
    X = df[features].values
    y = df['MEDV'].values

    return X, y

# 2) 目标函数: 给定超参数 x -> 评估分数 (我们要最大化验证集 R^2 或最小化 MSE)
# 注意：x 是 dict，如 {'n_estimators':100, 'max_depth':10, ...}
def evaluate_rf(X_train, y_train, X_val, y_val, params, random_state=0, cv=None):
    """
    训练随机森林并返回验证得分（用 R^2, 也返回 MSE）
    params: dict 包含参数 ['n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features']
    """
    # 拷贝 params 并转换适合 sklearn 的格式
    rf = RandomForestRegressor(
        n_estimators=int(params.get('n_estimators', 100)),
        max_depth=None if params.get('max_depth', None) is None else int(params.get('max_depth')),
        min_samples_split=int(params.get('min_samples_split', 2)),
        min_samples_leaf=int(params.get('min_samples_leaf', 1)),
        max_features=params.get('max_features', 'auto'),
        random_state=random_state,
        n_jobs=-1,
        bootstrap=params.get('bootstrap', True)
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_val)
    mse = mean_squared_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    return {'mse': mse, 'r2': r2, 'model': rf, 'pred': pred}

# 3) 代理模型：随机森林 surrogate，用来拟合超参数空间->得分
#    - 我们用 RandomForestRegressor 作为 surrogate（回归目标为验证 MSE 或 -r2）
#    - 并用随机候选优化采集函数（采样一批候选点，用 surrogate 估计 EI，然后挑选最优）

# 参数空间定义（用连续值/离散值）
param_space = {
    'n_estimators': (10, 500),          # integer
    'max_depth': (2, 50),               # integer, None 用特殊处理
    'min_samples_split': (2, 50),       # integer
    'min_samples_leaf': (1, 50),        # integer
    'max_features': (0.1, 1.0),         # float fraction of features
    'bootstrap': (0, 1)                 # 0 or 1 -> boolean
}

def sample_random_params(rng, n_samples=1):
    samples = []
    for _ in range(n_samples):
        s = {
            'n_estimators': int(rng.randint(param_space['n_estimators'][0], param_space['n_estimators'][1]+1)),
            'max_depth': int(rng.randint(param_space['max_depth'][0], param_space['max_depth'][1]+1)),
            'min_samples_split': int(rng.randint(param_space['min_samples_split'][0], param_space['min_samples_split'][1]+1)),
            'min_samples_leaf': int(rng.randint(param_space['min_samples_leaf'][0], param_space['min_samples_leaf'][1]+1)),
            'max_features': float(rng.uniform(param_space['max_features'][0], param_space['max_features'][1])),
            'bootstrap': bool(rng.randint(0,2))
        }
        # occasionally set max_depth to None (no limit)
        if rng.rand() < 0.05:
            s['max_depth'] = None
        samples.append(s)
    return samples

def params_to_vector(params):
    """把参数 dict 转成数值向量，便于 surrogate 训练。对于 None，我们用 -1 填充。"""
    vec = [
        params['n_estimators'],
        -1 if params['max_depth'] is None else params['max_depth'],
        params['min_samples_split'],
        params['min_samples_leaf'],
        params['max_features'],
        1 if params['bootstrap'] else 0
    ]
    return np.array(vec, dtype=float)

def vector_to_params(vec):
    p = {
        'n_estimators': int(round(vec[0])),
        'max_depth': None if vec[1] < 0 else int(round(vec[1])),
        'min_samples_split': int(round(vec[2])),
        'min_samples_leaf': int(round(vec[3])),
        'max_features': float(vec[4]),
        'bootstrap': bool(int(round(vec[5])))
    }
    # clamp ranges
    p['n_estimators'] = int(np.clip(p['n_estimators'], param_space['n_estimators'][0], param_space['n_estimators'][1]))
    if p['max_depth'] is not None:
        p['max_depth'] = int(np.clip(p['max_depth'], param_space['max_depth'][0], param_space['max_depth'][1]))
    p['min_samples_split'] = int(np.clip(p['min_samples_split'], param_space['min_samples_split'][0], param_space['min_samples_split'][1]))
    p['min_samples_leaf'] = int(np.clip(p['min_samples_leaf'], param_space['min_samples_leaf'][0], param_space['min_samples_leaf'][1]))
    p['max_features'] = float(np.clip(p['max_features'], param_space['max_features'][0], param_space['max_features'][1]))
    return p

# EI 采集函数（假设 surrogate 给出预测 mean & std）
def expected_improvement(mu, sigma, f_best, xi=0.01):
    # guard small sigma
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

# 4) 可选 PyTorch surrogate：一个小 MLP + MC-dropout，用来估计预测均值和不确定度
#    （训练方式：最小二乘，预测时多次前向取均值和方差）
class TorchSurrogateNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64,64), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_torch_surrogate(X, y, epochs=500, lr=1e-3, verbose=False):
    X_t = torch.tensor(X.astype(np.float32))
    y_t = torch.tensor(y.astype(np.float32))
    model = TorchSurrogateNet(input_dim=X.shape[1], hidden_dims=(128,128), dropout=0.15)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        pred = model(X_t)
        loss = F.mse_loss(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (ep % 100 == 0):
            print(f"Epoch {ep} loss {loss.item():.6f}")
    return model

def predict_torch_mc(model, X, n_samples=50):
    model.train()  # keep dropout on for MC dropout
    X_t = torch.tensor(X.astype(np.float32))
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = model(X_t).cpu().numpy()
            preds.append(p)
    preds = np.stack(preds, axis=0)  # (n_samples, n_points)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0, ddof=1)
    return mu, sigma

# 5) 贝叶斯优化主流程（surrogate = sklearn RF or PyTorch surrogate）
#    - 初始随机采样 n_init
#    - 每轮：训练 surrogate，产生一大批候选点（随机采样），用 surrogate 预测 mu/sigma，计算 EI，选择最大 EI 的点去真实评估
#    - 记录轨迹（每次最好分数、最佳超参）
def bayes_optimize_rf(X_train, y_train, X_val, y_val,
                      n_init=10, n_iter=40, surrogate_type='rf', random_seed=0,
                      candidate_pool_size=1000):
    """
    surrogate_type: 'rf' or 'torch'
    返回：history 列表，包含 (params, score, time)
    """
    rng = check_random_state(random_seed)
    # 数据容器
    vectors = []
    y_vals = []  # 我们用目标是 r2，若需要最小化 mse，可改
    params_list = []
    history = []

    # 初始随机点
    init_params = sample_random_params(rng, n_init)
    for p in init_params:
        res = evaluate_rf(X_train, y_train, X_val, y_val, p, random_state=random_seed)
        score = res['r2']  # 我们最大化 r2
        vectors.append(params_to_vector(p))
        y_vals.append(score)
        params_list.append(p)
        history.append({'params': p, 'r2': score, 'mse': res['mse']})

    best_idx = int(np.argmax(y_vals))
    best_score = y_vals[best_idx]
    best_params = params_list[best_idx].copy()

    for it in range(n_iter):
        # train surrogate on existing (vectors, y_vals)
        Xs = np.vstack(vectors)
        ys = np.array(y_vals)
        if surrogate_type == 'rf':
            # surrogate predicts mean by RF; to get sigma we use tree-wise predictions variance
            surf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_seed)
            surf.fit(Xs, ys)
            # candidate pool
            cands = sample_random_params(rng, n_samples=candidate_pool_size)
            cvecs = np.vstack([params_to_vector(c) for c in cands])
            mu = surf.predict(cvecs)
            # compute "sigma" as std of predictions from individual trees
            all_tree_preds = np.stack([t.predict(cvecs) for t in surf.estimators_], axis=0)  # (n_trees, n_cand)
            sigma = np.std(all_tree_preds, axis=0, ddof=1)
        else:
            # torch surrogate
            surf_model = train_torch_surrogate(Xs, ys, epochs=400, lr=1e-3, verbose=False)
            cands = sample_random_params(rng, n_samples=candidate_pool_size)
            cvecs = np.vstack([params_to_vector(c) for c in cands])
            mu, sigma = predict_torch_mc(surf_model, cvecs, n_samples=60)

        # compute EI (maximize)
        f_best = best_score
        ei = expected_improvement(mu, sigma, f_best, xi=0.01)
        # select best candidate
        idx_best = int(np.argmax(ei))
        cand_params = cands[idx_best]
        # evaluate real objective
        t0 = time.time()
        res = evaluate_rf(X_train, y_train, X_val, y_val, cand_params, random_state=random_seed)
        t1 = time.time()
        score = res['r2']
        # append
        vectors.append(params_to_vector(cand_params))
        y_vals.append(score)
        params_list.append(cand_params)
        history.append({'params': cand_params, 'r2': score, 'mse': res['mse'], 'time': t1 - t0})
        # update best
        if score > best_score:
            best_score = score
            best_params = cand_params.copy()
        # debug print
        print(f"Iter {it+1}/{n_iter} - cand best EI idx {idx_best}, r2={score:.4f}, best_r2={best_score:.4f}")

    return {'history': history, 'best_score': best_score, 'best_params': best_params}

# 6) 对照：随机搜索 baseline（相同评估次数）
def random_search_rf(X_train, y_train, X_val, y_val, n_evals=50, random_seed=0):
    rng = check_random_state(random_seed)
    history = []
    best_score = -np.inf
    best_params = None
    for i in range(n_evals):
        p = sample_random_params(rng, 1)[0]
        res = evaluate_rf(X_train, y_train, X_val, y_val, p, random_state=random_seed)
        score = res['r2']
        history.append({'params': p, 'r2': score, 'mse': res['mse']})
        if score > best_score:
            best_score = score
            best_params = p.copy()
        if (i+1) % 10 == 0:
            print(f"Random search eval {i+1}/{n_evals}, current best r2={best_score:.4f}")
    return {'history': history, 'best_score': best_score, 'best_params': best_params}

# 7) 主流程：读取数据 -> 划分 -> 标准化 -> 运行 BO (rf surrogate + torch surrogate) + 随机搜索
def run_full_experiment(seed=0):
    print("Generating dataset...")
    X, y = read_data('data.csv')
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # budgets
    n_init = 10
    n_iter = 40
    total_evals = n_init + n_iter

    print("Running Bayesian optimization with RF surrogate...")
    bo_rf = bayes_optimize_rf(X_train, y_train, X_val, y_val,
                              n_init=n_init, n_iter=n_iter, surrogate_type='rf', random_seed=seed,
                              candidate_pool_size=800)

    print("Running Bayesian optimization with Torch surrogate (MC-dropout)...")
    bo_torch = bayes_optimize_rf(X_train, y_train, X_val, y_val,
                                 n_init=n_init, n_iter=n_iter, surrogate_type='torch', random_seed=seed,
                                 candidate_pool_size=600)

    print("Running Random Search baseline...")
    rs = random_search_rf(X_train, y_train, X_val, y_val, n_evals=total_evals, random_seed=seed)

    # evaluate best on test set
    def eval_on_test(best_params):
        res = evaluate_rf(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]), X_test, y_test, best_params, random_state=seed)
        return res

    test_res_bo_rf = eval_on_test(bo_rf['best_params'])
    test_res_bo_torch = eval_on_test(bo_torch['best_params'])
    test_res_rs = eval_on_test(rs['best_params'])

    print("Test R2 - BO (RF surrogate):", test_res_bo_rf['r2'])
    print("Test R2 - BO (Torch surrogate):", test_res_bo_torch['r2'])
    print("Test R2 - Random Search:     ", test_res_rs['r2'])

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'bo_rf': bo_rf, 'bo_torch': bo_torch, 'rs': rs,
        'test_res_bo_rf': test_res_bo_rf, 'test_res_bo_torch': test_res_bo_torch, 'test_res_rs': test_res_rs
    }

# 8) 数据分析可视化
#    - A) 优化轨迹：每次评估的 best R2 随评估次数变化（对比 BO_rf, BO_torch, RS）
#    - B) surrogate 在参数空间上的预测 vs 真实观察（用第一个参数 n_estimators 做一维切片示意）
#    - C) 最终模型在测试集上的预测 vs 真值（预测曲线）
#    - D) 超参数散点图：展示评估点在两个重要参数维度上的分布及其得分（彩色可视化）
def plot_results(exp_res, savepath=None):
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)

    bo_rf = exp_res['bo_rf']
    bo_torch = exp_res['bo_torch']
    rs = exp_res['rs']
    X_test = exp_res['X_test']; y_test = exp_res['y_test']

    # A: 优化轨迹（累计最好 r2）
    def cum_best(history):
        bests = []
        cur_best = -np.inf
        for h in history:
            cur_best = max(cur_best, h['r2'])
            bests.append(cur_best)
        return np.array(bests)
    hist_bo_rf = bo_rf['history']
    hist_bo_torch = bo_torch['history']
    hist_rs = rs['history']

    cum_rf = cum_best(hist_bo_rf)
    cum_torch = cum_best(hist_bo_torch)
    cum_rs = cum_best(hist_rs)

    eval_idx_rf = np.arange(1, len(cum_rf)+1)
    eval_idx_torch = np.arange(1, len(cum_torch)+1)
    eval_idx_rs = np.arange(1, len(cum_rs)+1)


    plt.plot(eval_idx_rf, cum_rf, label='BO (RF surrogate)', linewidth=2)
    plt.plot(eval_idx_torch, cum_torch, label='BO (Torch surrogate)', linewidth=2)
    plt.plot(eval_idx_rs, cum_rs, label='Random Search', linewidth=2)
    plt.title("A) 优化轨迹：累计最好 R² 随评估次数变化（越高越好）")
    plt.xlabel("评估次数")
    plt.ylabel("累计最好 R²")
    plt.legend()
    plt.show()


    # B: surrogate slice - use BO_rf's surrogate final: fit RF surrogate on all evaluated points, then vary n_estimators (1D)
    all_vecs_rf = np.vstack([params_to_vector(h['params']) for h in hist_bo_rf])
    all_scores_rf = np.array([h['r2'] for h in hist_bo_rf])
    surf_final = RandomForestRegressor(n_estimators=300, random_state=0)
    surf_final.fit(all_vecs_rf, all_scores_rf)

    # vary n_estimators while holding others fixed at median
    med_vec = np.median(all_vecs_rf, axis=0)
    n_grid = np.linspace(param_space['n_estimators'][0], param_space['n_estimators'][1], 200)
    grid_vecs = []
    for n in n_grid:
        v = med_vec.copy()
        v[0] = n
        grid_vecs.append(v)
    grid_vecs = np.vstack(grid_vecs)
    mu_grid = surf_final.predict(grid_vecs)
    # get predictions from each tree to estimate sigma
    all_tree_preds = np.stack([t.predict(grid_vecs) for t in surf_final.estimators_], axis=0)
    sigma_grid = np.std(all_tree_preds, axis=0, ddof=1)

    plt.plot(n_grid, mu_grid, linestyle='-', linewidth=2)
    plt.fill_between(n_grid, mu_grid - 1.96*sigma_grid, mu_grid + 1.96*sigma_grid, alpha=0.25)
    plt.title("B) surrogate 对 n_estimators 的预测 (均值 ± 95% CI)")
    plt.xlabel("n_estimators")
    plt.ylabel("预测的 R²")
    plt.show()

    # C: final model predictions vs truth - use best BO_rf model
    best_params_rf = bo_rf['best_params']
    final_rf_model = RandomForestRegressor(
        n_estimators=int(best_params_rf['n_estimators']),
        max_depth=None if best_params_rf['max_depth'] is None else int(best_params_rf['max_depth']),
        min_samples_split=int(best_params_rf['min_samples_split']),
        min_samples_leaf=int(best_params_rf['min_samples_leaf']),
        max_features=best_params_rf['max_features'],
        random_state=0, n_jobs=-1, bootstrap=best_params_rf['bootstrap']
    )
    X_train_comb = np.vstack([exp_res['X_train'], exp_res['X_val']])
    y_train_comb = np.concatenate([exp_res['y_train'], exp_res['y_val']])
    final_rf_model.fit(X_train_comb, y_train_comb)
    y_pred_test = final_rf_model.predict(X_test)


    # sort by true y for a cleaner curve
    idx = np.argsort(y_test)
    plt.scatter(y_test[idx], y_pred_test[idx], s=12, alpha=0.8)
    # plot ideal line
    mn = min(y_test.min(), y_pred_test.min()); mx = max(y_test.max(), y_pred_test.max())
    plt.plot([mn, mx], [mn, mx], linestyle='--', linewidth=2)
    plt.title("C) 最终模型在测试集上的预测 vs 真值（越靠近对角线越好）")
    plt.xlabel("真值 y")
    plt.ylabel("预测 ŷ")
    plt.show()

    # D: hyperparam scatter (n_estimators vs max_depth) colored by r2
    param_points = np.array([params_to_vector(h['params']) for h in hist_bo_rf])
    scores = np.array([h['r2'] for h in hist_bo_rf])

    fig, ax3 = plt.subplots()
    sc = plt.scatter(param_points[:, 0], param_points[:, 1], c=scores, s=60, cmap='viridis')
    ax3.set_title("D) 超参数分布: n_estimators vs max_depth（颜色表示 R² 得分）")
    ax3.set_xlabel("n_estimators")
    ax3.set_ylabel("max_depth (None->negative)")
    cbar = fig.colorbar(sc, ax=ax3)
    cbar.set_label('R²')
    ax3.grid(True)
    plt.show()


    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()
    return fig

# 9) 主函数
if __name__ == "__main__":
    exp_res = run_full_experiment(seed=42)
    fig = plot_results(exp_res, savepath=None)