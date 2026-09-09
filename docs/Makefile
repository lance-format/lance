.PHONY: help serve build clean check-links sync make-full-website clean-full-website

help:
	@echo "Available commands:"
	@echo "  make serve        - Serve documentation locally (auto-installs deps)"
	@echo "  make build        - Build documentation (auto-installs deps)"
	@echo "  make make-full-website  - Assemble the full website from local repo checkouts"
	@echo "  make clean-full-website - Remove generated website content"
	@echo "                       Run 'make make-full-website' before 'make build' or 'make serve' for the full multi-repo site"
	@echo "                       Override any repo with env vars like LANCE_NAMESPACE_REPO=..., LANCE_SPARK_REPO=..., LANCE_RAY_REPO=..."
	@echo "  make clean        - Clean build artifacts"
	@echo "  make check-links  - Check for broken links"
	@echo "  make sync         - Sync dependencies with pyproject.toml"

sync:
	uv sync --all-extras

serve: make-full-website
	uv run mkdocs serve

build: make-full-website
	uv run mkdocs build

make-full-website:
	./make-full-website.sh

clean-full-website:
	./clean-full-website.sh

clean:
	rm -rf site/
	rm -rf .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +

check-links:
	uv run mkdocs-linkcheck src
