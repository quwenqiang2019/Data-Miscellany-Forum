FROM python:3.9.0

COPY . D:/workspace/github_proj/Data-Miscellany-Forum/src/深度学习模型部署-RESTfulAPI/keras-flask-deploy-webapp
WORKDIR D:/workspace/github_proj/Data-Miscellany-Forum/src/深度学习模型部署-RESTfulAPI/keras-flask-deploy-webapp
# Install dependencies
RUN pip install -r requirements.txt
# Run the application on port 5000
EXPOSE 5000
CMD ["python", "app.py"]
