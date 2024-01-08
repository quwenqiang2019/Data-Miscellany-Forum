import time
import aiohttp
import asyncio
import openpyxl
import itertools
import requests
import json


async def get_spec_names_onerepo(session, org, repo, br, access_token):
    url = f"https://gitee.com/api/v5/repos/{org}/{repo}/git/trees/{br}"
    headers = {'Authorization': f'Bearer {access_token}'}

    async with session.get(url, headers=headers) as response:
        error_repo_list = []
        try:
            # 获取 filetree
            res = await response.json()
            fileTree = []
            for file in res["tree"]:
                fileTree.append(file["path"])
            # 提取spec文件
            for file in fileTree:
                if file.endswith(".spec"):
                    specFile = file
                    break
            #
            # # 下载spec文件
            # url = f"https://gitee.com/{org}/{repo}/raw/{br}/{specFile}"
            # params = {
            #     "access_token": self.token
            # }
            # http = HTTPRequest(timeout=10)
            # status_code, response = await http.get(url, params)
            #
            # if status_code == 200:
            #     writeFileInDirectory(f"data/{org}/{br}/{repo}", specFile, response)
            #     return specFile
            #
            # # 写入数据库
            # specName = specFile
            # # print(specName)
            # if specName != None:
            #     Packages.update(specFile=specName).where(Packages.name == repo.name).execute()

            error_repo_list.append(None)
            return error_repo_list

        except:
            print(f"Error while fetching repositories {repo}")
            error_repo_list.append(repo)
            return error_repo_list




async def get_all_spec_names(org, br, access_token, repo_list):
    async with aiohttp.ClientSession() as session:
        tasks = [get_spec_names_onerepo(session, org, repo, br, access_token) for repo in repo_list]
        res = await asyncio.gather(*tasks)

        return res


async def main():
    org = 'src-oepkgs'
    br = "openEuler-22.03-LTS"
    access_token = 'xxxxxxxxxxxxxx'
    repo_list = ['esekeyd', 'execline', 'httest', 'psst']


    spec_names_repo = await get_all_spec_names(org, br, access_token, repo_list)
    print(spec_names_repo)


    # repository_names = [tup[0] for tup in repository_names_page]
    # repository_names = list(itertools.chain(*repository_names))
    #
    # pages = [tup[1] for tup in repository_names_page]
    # error_pages = list(itertools.chain(*pages))
    # error_pages = list(filter(None, error_pages))
    #
    #
    # print(error_pages)
    # start_time = time.time()  # 记录循环开始的时间
    # while error_pages and time.time() - start_time < 300:
    #     repository_names_page = await get_error_repository_names(org_name, access_token, error_pages)
    #     repository_names_error_page = [tup[0] for tup in repository_names_page]
    #     repository_names_error_page = list(itertools.chain(*repository_names_error_page))
    #     repository_names.extend(repository_names_error_page)
    #     pages = [tup[1] for tup in repository_names_page]
    #     error_pages = list(itertools.chain(*pages))
    #     error_pages = list(filter(None, error_pages))
    #
    # await write_to_excel(org_name, repository_names)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())