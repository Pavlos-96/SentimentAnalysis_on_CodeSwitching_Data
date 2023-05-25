
requirements_freeze:
	poetry export --dev --format requirements.txt > requirements.txt

install_kernel:
	poetry run python -m ipykernel install --user --name hi-en-sentiment --display-name "Python Hindi English Sentiment"

uninstall_kernel:
	jupyter kernelspec uninstall hi_en_sentiment