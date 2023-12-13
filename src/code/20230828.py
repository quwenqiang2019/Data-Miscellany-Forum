def province_distribution_of_colleges(self, file):
    df = pd.read_excel(os.path.join(self.datapath, file))

    df1 = df
    hua_bei = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区']
    dong_bei = ['辽宁省', '吉林省', '黑龙江省']
    hua_dong = ['上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省']
    hua_nan = ['广东省', '广西壮族自治区', '海南省']
    zhong_nan = ['湖南省', '湖北省', '河南省', '江西省']
    xi_nan = ['重庆市', '四川省', '贵州省', '云南省', '西藏自治区']
    xi_bei = ['陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区']
    gang_ao = ['香港特别行政区', '澳门特别行政区']

    df1['区域'] = None

    for index, row in df1.iterrows():
        if row['省份'] in hua_bei:
            df1.at[index, '区域'] = '华北'
        elif row['省份'] in dong_bei:
            df1.at[index, '区域'] = '东北'
        elif row['省份'] in hua_dong:
            df1.at[index, '区域'] = '华东'
        elif row['省份'] in hua_nan:
            df1.at[index, '区域'] = '华南'
        elif row['省份'] in zhong_nan:
            df1.at[index, '区域'] = '中南'
        elif row['省份'] in xi_nan:
            df1.at[index, '区域'] = '西南'
        elif row['省份'] in xi_bei:
            df1.at[index, '区域'] = '西北'
        elif row['省份'] in gang_ao:
            df1.at[index, '区域'] = '港澳'
        else:
            df1.at[index, '区域'] = '未知'

    print(df1)


province_distribution_of_colleges('schools_with_coordinates.xlsx')
