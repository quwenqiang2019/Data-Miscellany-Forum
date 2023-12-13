# 方法一：format方法
query = '''
    SELECT customer_id, COUNT(*) as num_orders
    FROM orders
    WHERE date >= '{start_date}' AND date <= '{end_date}'    
    GROUP BY customer_id
    HAVING num_orders > {min_orders}
'''

start_date=1
end_date=2
min_orders=3

formatted_query = query.format(start_date=start_date, end_date=end_date, min_orders=min_orders)
print (formatted_query)

# 方法二：%方法
query = '''
    SELECT salesperson, product, COUNT(*) AS num_sales
    FROM sales
    WHERE salesperson = %(salesperson)s      
    GROUP BY salesperson, product
    HAVING num_sales >= %(min_sales)s
    ORDER BY num_sales DESC;
'''
salesperson='a'
min_sales=10

formatted_query =query%{'salesperson': salesperson, 'min_sales': min_sales}
print (formatted_query)