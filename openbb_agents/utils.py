import os

from .models import AnsweredSubQuestion, SubQuestion


def get_verbosity() -> bool:
    return os.environ.get("VERBOSE", "False") == "True"


def get_dependencies(
    answered_subquestions: list[AnsweredSubQuestion], subquestion: SubQuestion
) -> list[AnsweredSubQuestion]:
    dependency_subquestions = [
        answered_subq
        for answered_subq in answered_subquestions
        if answered_subq.subquestion.id in (subquestion.depends_on or [])
    ]
    return dependency_subquestions
