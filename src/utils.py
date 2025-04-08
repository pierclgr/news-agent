import importlib
from datetime import datetime


def to_camel_case(snake_str):
    """Convert a snake_case string to CamelCase."""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def import_agent_class(agent_name):
    """Import a class from a module based on camel case conversion of module name."""
    # Convert agent_name to CamelCase for the class name
    class_name = to_camel_case(agent_name)

    # Construct the module path
    module_path = f"src.agents.{agent_name}"

    # Import the module dynamically
    module = importlib.import_module(module_path)

    # Get the class from the module
    agent_class = getattr(module, class_name)

    return agent_class


def get_today_date():
    return datetime.now().strftime("%Y-%m-%d")


def get_printable_articles_list(articles_list: list):
    title_idx = 2
    source_idx = 3
    date_idx = 4

    output = ""
    for index, row in enumerate(articles_list):
        article_number_str = f"{index}) "
        output += f"\t{article_number_str}{row[title_idx]} - {row[source_idx]}, {row[date_idx]}\n"

    return output
