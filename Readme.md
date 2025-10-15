docker-compose up -d

docker exec -it cvat_server /bin/bash

# From inside the container's shell, run the following command 
# and follow the prompts to set a username, email, and password
python3 manage.py createsuperuser

exit

http://localhost:8080

