# 解决每次session 在git-push时都需要输入密码的问题
eval $(ssh-agent)
ssh-add ~/.ssh/id_rsa
