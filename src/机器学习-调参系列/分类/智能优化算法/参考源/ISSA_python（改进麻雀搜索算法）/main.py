from ISSA import ISSA
from SSA import SSA
import matplotlib.pyplot as plt
import test_function

if __name__ == '__main__':

    n_dim = 30
    lb = [-100 for i in range(n_dim)]
    ub = [100 for i in range(n_dim)]
    demo_func = test_function.fu1
    pop_size = 30
    max_iter = 500
    # 运行SSA算法
    ssa = SSA(demo_func, n_dim=n_dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub)
    ssa.run()
    print('best_x_ssa is ', ssa.gbest_x, 'best_y is', ssa.gbest_y)

    # 运行SSA算法
    issa = ISSA(demo_func, n_dim=n_dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub)
    issa.run()
    print('best_x_issa is ', issa.gbest_x, 'best_y_issa is', issa.gbest_y)
    # print(f'{demo_func(ssa.gbest_x)}\t{ssa.gbest_x}')
    # 绘制收敛曲线
    plt.semilogy(ssa.gbest_y_hist, label='SSA')
    plt.semilogy(issa.gbest_y_hist, label='ISSA')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Convergence Plot')
    plt.legend()
    # plt.grid(True)
    plt.show()
