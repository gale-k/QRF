from relbench.base import Table, Database, Dataset, EntityTask
from relbench.datasets import get_dataset
from relbench.tasks import get_task


dataset = get_dataset("rel-stack", download=True)
print(f"dataset: {dataset}\n\n")

db = dataset.get_db()
print(f"database: {db}\n\n")

task = get_task("rel-stack", "user-engagement", download=True)
print(f"task: {task}\n\n")

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

print(f"train_table: {train_table}\n\n")
print(f"val_table: {val_table}\n\n")
print(f"test_table: {test_table}\n\n")

# table data:
# user_id   contribution (bool) 

# task.evaluate(test_pred)

# task.evaluate(val_pred, val_table)