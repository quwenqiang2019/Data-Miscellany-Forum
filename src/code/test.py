from pyrpm.spec import Spec, replace_macros

def getURLandSourceFromSpecFileOneRepo(spec_file):
    url, source = None, None


    spec = Spec.from_file(spec_file)
    print(spec)
    if spec.sources == [] and spec.url == '':
        url, source = None, None

    if spec.url is not None:
        urlList = extractURLFromStr(replace_macros(spec.url, spec))
        if urlList != []:
            url = urlList[0]
            print(url)
        # if spec.sources != []:
        #     sourceList = extractURLFromStr(replace_macros(spec.sources[0], spec))
        #     if sourceList != []:
        #         source = sourceList[0]
        # if url is None:
        #     if len(spec.macros) >= 1:
        #         if spec.macros.get('goipath'):
        #             url = "https://" + spec.macros['goipath'] + "/"
        #
        # return url, source




if __name__ == '__main__':
    spec_file = 'D:\workspace\gitlab_proj\\upstream-observer\data\src-oepkgs\openEuler-22.03-LTS\ghc-language-ecmascript\ghc-language-ecmascript.spec'
    getURLandSourceFromSpecFileOneRepo(spec_file)
