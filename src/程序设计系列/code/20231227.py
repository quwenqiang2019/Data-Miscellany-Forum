import aiohttp
import asyncio
import openpyxl

async def fetch_repositories(session, org_name, page=1):
    url = f'https://gitee.com/api/v5/orgs/{org_name}/repos'
    params = {'page': page}
    async with session.get(url, params=params) as response:
        return await response.json()

async def write_to_excel(repositories):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Repository Name', 'Description', 'URL'])

    for repo in repositories:
        ws.append([repo['name'], repo['description'], repo['html_url']])

    wb.save('gitee_repositories.xlsx')

async def main():
    org_name = 'src-oepkgs'
    access_token = 'xxxxxxxxxxxxxx'

    headers = {'Authorization': f'Bearer {access_token}'}

    all_repositories = []
    page = 1
    max_pages = 3   # 获取前n页
    # target_pages = [8, 15, 55]  # 想要获取的页码列表

    async with aiohttp.ClientSession(headers=headers) as session:
        # while True:
        while page <= max_pages:
        # for page in target_pages:
            repositories = await fetch_repositories(session, org_name, page)
            if not repositories:
                break

            all_repositories.extend(repositories)
            page += 1

            print(page)

    await write_to_excel(all_repositories)
    print('Repositories information written to gitee_repositories.xlsx')

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # if not asyncio.get_event_loop().is_closed():
    #     asyncio.run(main())
    # else:
    #     asyncio.set_event_loop(asyncio.new_event_loop())
    #     asyncio.run(main())





