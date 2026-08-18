class BankAccount:
    # 类属性：所有账户共享
    bank_name = "宇宙银行"
    interest_rate = 0.03        # 年利率 3%
    _total_accounts = 0         # 账户总数（私有类属性）
    
    def __init__(self, owner, balance=0):
        self.owner = owner      # 实例属性：户主
        self.balance = balance  # 实例属性：余额
        BankAccount._total_accounts += 1
        self.account_id = f"ACC-{BankAccount._total_accounts:06d}"
    
    # ========== 实例方法：操作具体账户 ==========
    def deposit(self, amount):
        """存钱——只能对具体账户操作"""
        if amount <= 0:
            raise ValueError("存款金额必须大于0")
        self.balance += amount
        return self
    
    def withdraw(self, amount):
        """取钱——只能对具体账户操作"""
        if amount > self.balance:
            raise ValueError("余额不足")
        self.balance -= amount
        return self
    
    def transfer(self, other, amount):
        """转账——涉及两个具体账户"""
        self.withdraw(amount)
        other.deposit(amount)
        return self
    
    def info(self):
        """查看账户信息"""
        return f"[{self.account_id}] {self.owner}: ¥{self.balance:.2f}"
    
    # ========== 类方法：操作类级别数据 ==========
    @classmethod
    def get_total_accounts(cls):
        """查看开了多少个账户——不依赖某个具体账户"""
        return cls._total_accounts
    
    @classmethod
    def set_interest_rate(cls, new_rate):
        """央行调整利率——影响所有账户"""
        cls.interest_rate = new_rate
        print(f"利率已调整至 {new_rate*100}%")
    
    @classmethod
    def create_savings_account(cls, owner, initial_deposit):
        """工厂方法：创建储蓄账户（带最低存款要求）"""
        if initial_deposit < 1000:
            raise ValueError("储蓄账户最低存款 ¥1000")
        return cls(owner, initial_deposit)
    
    # ========== 静态方法：独立工具函数 ==========
    @staticmethod
    def validate_id_card(id_number):
        """验证身份证号——和任何账户/类数据无关"""
        if len(id_number) != 18:
            return False
        # 简化校验：最后一位校验码
        weights = [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
        check_codes = "10X98765432"
        try:
            total = sum(int(id_number[i]) * weights[i] for i in range(17))
            return check_codes[total % 11] == id_number[17].upper()
        except ValueError:
            return False
    
    @staticmethod
    def format_currency(amount):
        """格式化金额——纯工具函数"""
        return f"¥{amount:,.2f}"


# ========== 实际使用 ==========

# 1. 实例方法：操作具体账户
alice = BankAccount("Alice", 5000)
bob = BankAccount("Bob", 3000)

alice.deposit(2000).withdraw(500)   # 链式调用
alice.transfer(bob, 1000)
print(alice.info())   # [ACC-000001] Alice: ¥5500.00
print(bob.info())     # [ACC-000002] Bob: ¥4000.00

# 2. 类方法：不创建账户也能查/改类级别信息
print(BankAccount.get_total_accounts())   # 2
BankAccount.set_interest_rate(0.035)      # 利率已调整至 3.5%

# 通过实例也能调用类方法（但不常见）
print(bob.get_total_accounts())           # 2

# 工厂方法创建账户
charlie = BankAccount.create_savings_account("Charlie", 5000)

# 3. 静态方法：完全独立的工具，只是挂在类下面
print(BankAccount.validate_id_card("11010519491231002X"))  # True
print(BankAccount.format_currency(1234567.89))             # ¥1,234,567.89

# 实例也能调静态方法（但不推荐，语义上容易混淆）
print(alice.format_currency(100))         # ¥100.00