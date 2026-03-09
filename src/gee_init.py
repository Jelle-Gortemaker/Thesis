import ee

def init_gee():
    ee.Initialize(project="thesis-489710")
    print("GEE initialized.")