import time
import aiohttp
import asyncio
import openpyxl
import itertools
import requests

# policy = asyncio.WindowsSelectorEventLoopPolicy()
# asyncio.set_event_loop_policy(policy)
# semaphore = asyncio.Semaphore(16)

async def get_repository_names_onepage(session, org_name, access_token, page):
    url = f'https://gitee.com/api/v5/orgs/{org_name}/repos'
    params = {'page': page}
    headers = {'Authorization': f'Bearer {access_token}'}


    async with session.get(url, headers=headers, params=params) as response:

        error_page_list = []
        try:
            repositories = await response.json()
            error_page_list.append(None)
            return [repo['name'] for repo in repositories], error_page_list
        except:
            print(f"Error while fetching repositories {page}")
            error_page_list.append(page)
            return [], error_page_list



async def get_all_repository_names(org_name, access_token, num_pages):
    async with aiohttp.ClientSession() as session:
        tasks = [get_repository_names_onepage(session, org_name, access_token, page) for page in range(1, num_pages + 1)]
        res = await asyncio.gather(*tasks)

        return res

async def get_error_repository_names(org_name, access_token, error_pages):
    async with aiohttp.ClientSession() as session:
        tasks = [get_repository_names_onepage(session, org_name, access_token, page) for page in error_pages]
        res = await asyncio.gather(*tasks)

        return res


async def get_total_pages(org_name, access_token):
    url = f'https://gitee.com/api/v5/orgs/{org_name}/repos'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    print(response.headers)
    total_pages = int(response.headers.get("total_page"))
    total_counts = int(response.headers.get("total_count"))

    return total_pages, total_counts


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
    access_token = 'ed1d0fb3d6aa397c514569b4b965e3a9'
    num_pages, total_repo = await get_total_pages(org_name, access_token)
    print(num_pages)
    print(total_repo)

    repository_names_page = await get_all_repository_names(org_name, access_token, num_pages)
    repository_names = [tup[0] for tup in repository_names_page]
    repository_names = list(itertools.chain(*repository_names))

    pages = [tup[1] for tup in repository_names_page]
    error_pages = list(itertools.chain(*pages))
    error_pages = list(filter(None, error_pages))


    print(error_pages)
    start_time = time.time()  # 记录循环开始的时间
    while error_pages and time.time() - start_time < 300:
        repository_names_page = await get_error_repository_names(org_name, access_token, error_pages)
        repository_names_error_page = [tup[0] for tup in repository_names_page]
        repository_names_error_page = list(itertools.chain(*repository_names_error_page))
        repository_names.extend(repository_names_error_page)
        pages = [tup[1] for tup in repository_names_page]
        error_pages = list(itertools.chain(*pages))
        error_pages = list(filter(None, error_pages))

    await write_to_excel(org_name, repository_names)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # if not asyncio.get_event_loop().is_closed():
    #     asyncio.run(main())
    # else:
    #     asyncio.set_event_loop(asyncio.new_event_loop())
    #     asyncio.run(main())




