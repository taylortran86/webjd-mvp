# Install dependencies inside Poetry environment
install:
	poetry install

# Run the backend server (adjust path as needed)
run:
	poetry run python src/legal_chatbot_test/app.py

# Clean cache and temporary files
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache

.PHONY: test-endpoint

test-endpoint:
	curl -X $(type) http://127.0.0.1:5000/$(endpoint) \
	     -H "Content-Type: application/json" \
	     -d '{"message": "hello"}'
