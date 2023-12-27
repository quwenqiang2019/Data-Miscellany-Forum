import aiohttp
import asyncio
import openpyxl
import itertools
import requests

# policy = asyncio.WindowsSelectorEventLoopPolicy()
# asyncio.set_event_loop_policy(policy)
# semaphore = asyncio.Semaphore(16)

async def fetch_repositories(session, org_name, page):
    url = f'https://gitee.com/api/v5/orgs/{org_name}/repos'
    params = {'page': page}
    access_token = 'ed1d0fb3d6aa397c514569b4b965e3a9'
    headers = {'Authorization': f'Bearer {access_token}'}
    async with session.get(url, headers = headers, params=params) as response:

        content_type = response.headers.get('Content-Type', '').lower()
        data = await response.read()

        if 'application/json' in content_type:
            return await response.json(), response.headers.get('Link')
        elif 'text/plain' in content_type:
            raise ValueError(f"Unexpected content type: {content_type}. Response content: {data.decode('utf-8')}")
        else:
            raise ValueError(f"Unsupported content type: {content_type}. Response content: {data.decode('utf-8')}")


async def get_repository_names(session, org_name, page):
    error_page_list = []

    try:
        repositories, _ = await fetch_repositories(session, org_name, page)
        error_page_list.append(None)
        return [repo['name'] for repo in repositories], error_page_list
    except ValueError as e:
        print(f"Error while fetching repositories {page}: {e}")
        error_page_list.append(page)
        return [], error_page_list

async def get_all_repository_names(org_name, num_pages):
    async with aiohttp.ClientSession() as session:
        tasks = [get_repository_names(session, org_name, page) for page in range(1, num_pages + 1)]

        return await asyncio.gather(*tasks)


async def get_total_pages(org_name):

    url = f'https://gitee.com/api/v5/orgs/{org_name}/repos'

    access_token = 'ed1d0fb3d6aa397c514569b4b965e3a9'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    total_pages = int(response.headers.get("total_page"))

    return total_pages


async def write_to_excel(org_name, repository_names):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Repository Name'])

    for name in repository_names:
        ws.append([str(name)])  # Convert to string before appending

    excel_filename = f'{org_name}_repositories.xlsx'
    wb.save(excel_filename)
    print(f'Repositories information written to {excel_filename}')


async def main():
    org_name = 'src-oepkgs'
    # num_pages = 18  # 请根据需要调整获取的页数
    num_pages = await get_total_pages(org_name)
    print(num_pages)

    repository_names = await get_all_repository_names(org_name, num_pages)   # 获取所有仓库名称，一个二维列表，每一个元素是一页的仓库名
    # print(repository_names)
    repository_names = list(itertools.chain(*repository_names))
    await write_to_excel(org_name, repository_names)



if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # if not asyncio.get_event_loop().is_closed():
    #     asyncio.run(main())
    # else:
    #     asyncio.set_event_loop(asyncio.new_event_loop())
    #     asyncio.run(main())




