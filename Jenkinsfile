pipeline {
    agent any

    environment {
        DOCKERHUB_USER = "yash450987"
        IMAGE_NAME = "yaswanth_2022bcs0049_lab4"
    }

    stages {

        stage('Checkout') {
            steps {
                git branch: 'main',
                url: 'https://github.com/2022bcs0049-yaswanthcv/Lab4.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t $DOCKERHUB_USER/$IMAGE_NAME:latest -t $DOCKERHUB_USER/$IMAGE_NAME:${BUILD_NUMBER} ."
            }
        }

        stage('Login to DockerHub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'USER',
                    passwordVariable: 'PASS'
                )]) {
                    sh '''
                    echo $PASS | docker login -u $USER --password-stdin
                    '''
                }
            }
        }

        stage('Push Image') {
            steps {
                sh "docker push $DOCKERHUB_USER/$IMAGE_NAME:latest"
                sh "docker push $DOCKERHUB_USER/$IMAGE_NAME:${BUILD_NUMBER}"
            }
        }
    }
}
