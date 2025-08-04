pipeline {
    agent any

    stages {
        stage('Install uv (if needed)') {
            steps {
                sh '''
                    if ! command -v uv &> /dev/null
                    then
                        echo "uv not found, installing..."
                        curl -Ls https://astral.sh/uv/install.sh | bash
                        export PATH="$HOME/.cargo/bin:$PATH"
                    fi
                '''
            }
        }


        stage('Run model training') {
            steps {
                // Falls das Skript z. B. scripts/model_training.py heißt:
                sh 'uv run python scripts/model_training.py'
            }
        }
    }
}
