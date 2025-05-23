To build the Docker image run:
```
docker build -f app/Dockerfile -t chestmodal .
```


To run the Docker image type:
```
docker run -p 8000:8000 chestmodal
```

This model deployment can be accessed at

[http://localhost:8000/](http://localhost:8000/)



To see the documentation and test out the endpoints, visit:

[http://localhost:8000/docs](http://localhost:8000/docs)
