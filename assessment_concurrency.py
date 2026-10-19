"""Bounded assessment admission and per-student exclusion for one API process."""
from contextlib import contextmanager
import os
import threading


def assessment_capacity():
    value = os.environ.get("MANDARIN_ASSESSMENT_CONCURRENCY", "2")
    try:
        capacity = int(value)
    except ValueError as exc:
        raise ValueError("MANDARIN_ASSESSMENT_CONCURRENCY must be a positive integer") from exc
    if capacity < 1:
        raise ValueError("MANDARIN_ASSESSMENT_CONCURRENCY must be a positive integer")
    return capacity


class AssessmentBusy(TimeoutError):
    pass


class AssessmentCoordinator:
    def __init__(self, capacity, wait_seconds=30):
        if capacity < 1:
            raise ValueError("Assessment capacity must be positive")
        self.capacity = capacity
        self.wait_seconds = wait_seconds
        self._condition = threading.Condition()
        self._active = 0
        self._students = set()

    @contextmanager
    def _operation(self, student_id, needs_slot):
        with self._condition:
            admitted = self._condition.wait_for(
                lambda: student_id not in self._students
                and (not needs_slot or self._active < self.capacity),
                timeout=self.wait_seconds,
            )
            if not admitted:
                raise AssessmentBusy("Assessment capacity is busy. Please retry shortly.")
            self._students.add(student_id)
            if needs_slot:
                self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._students.remove(student_id)
                if needs_slot:
                    self._active -= 1
                self._condition.notify_all()

    def assessment(self, student_id):
        """Limit expensive work; a student cannot assess and calibrate simultaneously."""
        return self._operation(student_id, needs_slot=True)

    def student(self, student_id):
        """Wait for this student's operations before deletion, without blocking others."""
        return self._operation(student_id, needs_slot=False)
