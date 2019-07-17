def git_init():
    with open('access_token.txt', 'r') as file:
        github_access_token = file.read() #create token from https://github.com/settings/tokens and add it to access_token.txt
    from github import Github
    g = Github(github_access_token)
    return g

def download_repo(path, clone_url):
    from git import Repo
    cloned_repo = Repo.clone_from(clone_url, path)
    assert cloned_repo.__class__ is Repo  # clone an existing repository

import os
def download_repositories(query, number=1):
    g = git_init()
    count = 0
    if not os.path.isdir('repos/'):
        os.makedirs('repos/')
    for repo in g.search_repositories(query):
        if repo.language=='Python':
            print(repo.name, repo.clone_url)
            dirname = 'repos/'+repo.name
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
                download_repo(dirname, repo.clone_url)
            count += 1
            if count>=number:
                break
            
download_repositories("TheAlgorithms")