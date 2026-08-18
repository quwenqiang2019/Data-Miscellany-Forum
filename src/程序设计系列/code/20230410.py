
import re

str = "https://api.github.com/repos/jimitpatel3699/Single_Page_App"
s_list = re.findall('repos/(.+)',str)
print(s_list)