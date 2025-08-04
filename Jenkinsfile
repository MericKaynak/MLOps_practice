pipeline {
    agent any

    stages {
        // Dieser Stage ist implizit und wird von Jenkins am Anfang ausgeführt
        // stage('Checkout') {
        //     steps {
        //         checkout scm
        //     }
        // }

        stage('Install Dependencies & Pull Data') {
            steps {
                sh '''
                    # Installiert uv, falls es nicht vorhanden ist
                    if ! command -v uv &> /dev/null
                    then
                        echo "uv not found, installing..."
                        curl -Ls https://astral.sh/uv/install.sh | bash
                        # Wichtig: Den Pfad für den Rest des Skripts aktualisieren
                        export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
                    fi

                    # Schritt 1: DVC installieren
                    # Wir verwenden uv, um dvc zu installieren.
                    echo "Installing DVC..."
                    uv pip install dvc

                    # Schritt 2: Die von DVC verwalteten Daten herunterladen
                    echo "Pulling data with DVC..."
                    dvc pull

                    # Überprüfen, ob die Datei jetzt da ist (optional, aber gut zum Debuggen)
                    echo "Verifying data file:"
                    ls -l data/
                '''
            }
        }

        stage('Run model training') {
            steps {
                sh '''
                    # Sicherstellen, dass der PATH für diesen neuen Shell-Block korrekt ist
                    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

                    echo "Starting model training..."
                    uv run python scripts/model_training.py
                '''
            }
        }
    }
}