from nicegui import ui

def build_ui():
    global orgText, countText, grid
    orgText = ui.label(getOrgAndBranchText())
    countText = ui.label(getRepoCountText())

    tableData = getTableData()
    grid = ui.aggrid({
        'defaultColDef': {'flex': 1},
        'columnDefs': [
            {'headerName': 'ID', 'field': 'id', 'checkboxSelection': True, 'flex': 1},
            {'headerName': '仓库名', 'field': 'name', 'filter': 'agTextColumnFilter', 'floatingFilter': True, 'flex': 2},
            {'headerName': 'SPEC 版本', 'field': 'specVersion', 'filter': 'agTextColumnFilter', 'floatingFilter': True,
            'flex': 2},
            {'headerName': '上游版本', 'field': 'upstreamVersion', 'filter': 'agTextColumnFilter', 'floatingFilter': True,
            'flex': 2},
            {'headerName': '上游地址', 'field': 'upstreamURL', 'filter': 'agTextColumnFilter', 'floatingFilter': True,
            'flex': 4},
            {'headerName': '采集器', 'field': 'method', 'filter': 'agTextColumnFilter', 'floatingFilter': True, 'flex': 1},
            {'headerName': '上次更新', 'field': 'lastUpdate', 'filter': 'agTextColumnFilter', 'floatingFilter': True,
            'flex': 2},
        ],
        'rowData': tableData,
        'rowSelection': 'multiple',
    }).style('height: 600px')

    ui.separator()
    with ui.row():
        with ui.card():
            with ui.row():
                with ui.button('抓取数据索引', on_click=fetchDataList):
                    ui.tooltip('抓取当前组织下的仓库列表').classes('bg-grey text-body2')
                with ui.button('抓取 spec 文件', on_click=fetchSpecFile):
                    ui.tooltip('抓取当前数据库中所有包的 spec 文件').classes('bg-grey text-body2')
                with ui.button('解析 spec 文件', on_click=fetchRepoData):
                    ui.tooltip('解析 spec 文件中的上游信息、版本号').classes('bg-grey text-body2')

        with ui.card():
            with ui.row():
                with ui.button('抓取软件包上游数据', on_click=fetchUpstreamData).classes('bg-green'):
                    ui.tooltip('从上游抓取软件包的版本号').classes('bg-grey text-body2')
                with ui.button('搜索未知软件包', on_click=searchUnknownPackage).classes('bg-green'):
                    ui.tooltip('搜索未抓取到的软件的上游').classes('bg-grey text-body2')

        with ui.card():
            with ui.row():
                ui.button('导出 nvchecker TOML', on_click=exportNvchecker).classes('bg-teal')
                ui.button('导出 openEuler Advisor YAML', on_click=exportOpenEuler).classes('bg-teal')

        with ui.card():
            with ui.row():
                ui.button('刷新数据', on_click=updateUIText).classes('bg-grey')



def getOrgAndBranchText():
    pass

def getRepoCountText():
    pass

def getTableData():
    pass

def fetchDataList():
    pass

def fetchSpecFile():
    pass

def fetchRepoData():
    pass

def fetchUpstreamData():
    pass

def searchUnknownPackage():
    pass

def exportNvchecker():
    pass

def exportOpenEuler():
    pass

def updateUIText():
    pass


if __name__ in {"__main__", "__mp_main__"}:
    build_ui()
    ui.run(host="127.0.0.1", port=8081,
    title='Upstream-Observer GUI', favicon='🍜', language='zh-CN')