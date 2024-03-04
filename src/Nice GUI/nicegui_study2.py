import nicegui as ng

def calculate(event):
    num1 = float(input1.get())
    num2 = float(input2.get())
    operator = operator_selection.get()

    if operator == '+':
        result = num1 + num2
    elif operator == '-':
        result = num1 - num2
    elif operator == '*':
        result = num1 * num2
    elif operator == '/':
        if num2 != 0:
            result = num1 / num2
        else:
            result = "Error: Division by zero"
    else:
        result = "Invalid operator"

    result_label.set(str(result))

app = ng.App(title="简单计算器", size=(300, 200))

with ng.Col():
    ng.Label(text="输入数字:")
    input1 = ng.Input(type="number")
    input2 = ng.Input(type="number")
    ng.Label(text="选择操作符:")
    operator_selection = ng.Dropdown(values=["+", "-", "*", "/"])
    result_label = ng.Label()
    ng.Button(text="计算", on_click=calculate)

app.run()
