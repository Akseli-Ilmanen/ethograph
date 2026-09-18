import ethograph as eto

project = eto.spot.Project(r"C:\Users\aksel\Documents\Code\ethograph\projects\paper\spot\spot.yaml")
project.materialise()  # every trial's video -> frames/ + E2E-Spot's index; resumable
result = project.train()  # runs/ctx2s_res…ms/
metrics = project.evaluate()  # ses-03, never trained on -> test_metrics.yaml

print(result.run_dir)
print(metrics)  # per class: misses, spurious, error in ms, hit rate at 10/20/50/100 ms
