pipeline {
    agent any

    environment {
        PYTHONUNBUFFERED = '1'
        PYTHONPATH = 'src'
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install -e .
                '''
            }
        }

        stage('Run Unit & Robustness Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest -v --junitxml=reports/junit_test_results.xml
                '''
            }
        }

        stage('Run Evaluation & Benchmarking') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python -m sentiment_qa.evaluate --dataset datasets/sentiment_cases.json --report reports/evaluation.json
                '''
            }
        }

        stage('Run DeepEval LLM Evaluation') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python -m sentiment_qa.deepeval_adapter --dataset datasets/sentiment_cases.json
                '''
            }
        }

        stage('Run Security Red-Teaming Audit') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python -m sentiment_qa.red_team
                '''
            }
        }
    }

    post {
        always {
            // Archive HTML Dashboard & JSON Evaluation Reports in Jenkins
            archiveArtifacts artifacts: 'reports/**/*.html, reports/**/*.json', allowEmptyArchive: true
            
            // Publish JUnit Test Results in Jenkins UI
            junit testResults: 'reports/junit_test_results.xml', allowEmptyResults: true
        }
        success {
            echo '--- Sentiment AI QA Pipeline Completed Successfully ---'
        }
        failure {
            echo '--- Sentiment AI QA Pipeline Failed ---'
        }
    }
}
