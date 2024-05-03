'''
Create a singleton decorator to ensure that for classes like Earth, Satellite, and RadarSystem,
only one instance exists throughout the project.
Declaring these classes as singletons prevents the often hard-to-detect hidden bugs that can arise from inadvertently
creating multiple instances which may lead to unsynchronized data.
Moreover, it simplifies the usage of these classes for other parts of the project.
For example, when declaring instances such as earth1 = Earth() and earth2 = Earth(), both earth1 and earth2 in storage
will actually point to the same object, thanks to the singleton pattern.
'''
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance