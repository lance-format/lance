def pytest_addoption(parser):
    parser.addoption(
        "--update-outputs",
        action="store_true",
        help="Rewrite the recorded snippet outputs instead of checking them.",
    )
