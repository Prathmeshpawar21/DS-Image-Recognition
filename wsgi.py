from app import app


if __name__ == '__main__':
    try:
        # app.run(host='0.0.0.0', port=8000)  # for gunicorn deployment purpose
        app.run(debug=True , host='0.0.0.0', port=5000)  # for Localhost
    except Exception as e:
        logging.error(f"Error running app: {e}")