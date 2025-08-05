pipeline {
    agent any

    stages {
        // Dieser Stage ist implizit und wird von Jenkins am Anfang ausgeführt
        // stage('Checkout') {
        //     steps {
        //         checkout scm
        //     }
        // }

        stage('Get Data') {
            steps {
                sh '''
                    echo "Starting data ingestion"
                    uv run python scripts/ingestion.py
                '''
            }
        }

        stage('Run model training') {
            steps {
                sh '''
                    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

                    echo "Starting model training..."
                    uv run python scripts/model_training.py
                '''
            }
        }
    }
}