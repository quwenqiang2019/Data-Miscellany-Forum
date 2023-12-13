import pandas as pd
import json

# 假设给定的JSON数据已经存储在data变量中
data = [
    {
        "title": "Data Source Adapter for Excel Sheets",
        "project_code_url": "https://github.com/polypheny/Polypheny-DB/pull/418",
        "date_created": "2022-05-17T23:30:01.526934Z",
        "tech_tags": [
            "java",
            "typescript"
        ],
        "topic_tags": [
            "database"
        ],
        "status": "passed",
        "program_slug": "2022",
        "contributor_display_name": "Kelly Xie",
        "mentor_names": [
            "Marc Hennemann",
            "Isabel"
        ],
        "abstract_short": "This project will allow Polypheny to interact with Excel sheets by adding a data source adapter. The Excel adapter enables Polypheny to query the...",
        "abstract_html": "This project will allow Polypheny to interact with Excel sheets by adding a data source adapter. \nThe Excel adapter enables Polypheny to query the mapped data using available query languages of Polypheny-DB and the imported tables can be joined with other tables.",
        "date_archived": "2022-05-17T23:30:01.526934Z",
        "id": "axdeCi5w",
        "organization_name": "Polypheny",
        "organization_slug": "polypheny"
    },
    {
        "title": "Admin Web Portal: New Features Support and Spam Mitigation",
        "project_code_url": "https://docs.google.com/document/d/1KiEZaYkCz7olJ5OeUJKcmuRn1C0qiUJ4UaHvesbcmgc/edit?usp=sharing",
        "date_created": "2022-05-17T23:30:01.954880Z",
        "tech_tags": [
            "node.js",
            "typescript"
        ],
        "topic_tags": [
            "New Features Support",
            "Spam Mitigation"
        ],
        "status": "passed",
        "program_slug": "2022",
        "contributor_display_name": "Asmit Kumar Sirohi",
        "mentor_names": [
            "Yasharth Dubey",
            "Jason Gayle"
        ],
        "abstract_short": "My idea for this GSoC period is basically about improving the UI/UX of the admin portal and making it according to the design standards that are...",
        "abstract_html": "My idea for this GSoC period is basically about improving the UI/UX of the admin portal and making it according to the design standards that are defined in talawa docs, also I will make it mobile responsive. Another focus I have this summer is to make all the screens (components) of talawa-admin functional i.e. No more hard-coded values in the admin portal, all the data will be live from talawa-API. I will also implement a way so that users can select or use talawa-admin in their preferred language and I will be implementing a feature for detecting whether a user is spamming a chat or not. Below are the features for talawa-admin that I am going to add this summer that will boost its usability, user experience, and its use cases.\n\nTalawa-admin Features: \n\nInteractive UI/UX.\nMultiple screens (Mobile or Tablet) are responsive.\nFunctional screens (components).\nImplementing the support for different languages.\nFeature to detect whether the user is spamming the chat or not.\nMigration from redux-routing to react-routing. *",
        "date_archived": "2022-05-17T23:30:01.954880Z",
        "id": "hMUkWQlA",
        "organization_name": "The Palisadoes Foundation",
        "organization_slug": "the-palisadoes-foundation"
    }
]

# 提取所需字段
df_data = []
for d in data:
    df_data.append({
        'title': d['title'],
        'project_code_url': d['project_code_url'],
        'tech_tags': d['tech_tags'],
        'topic_tags': d['topic_tags'],
        'status': d['status'],
        'contributor_display_name': d['contributor_display_name'],
        'mentor_names': d['mentor_names'],
        'id': d['id'],
        'organization_name': d['organization_name']
    })

# 转换为DataFrame
df = pd.DataFrame(df_data)

# 查看结果
print(df)
