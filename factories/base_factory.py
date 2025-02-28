class BaseFactory:
    REGISTRY = {}
    ITEM_KEY = ''

    @classmethod
    def _validate_and_instantiate(cls, item_name, *args, **kwargs):
        if item_name not in cls.REGISTRY:
            raise ValueError(f"Invalid item: {item_name}")

        config = cls.REGISTRY[item_name]

        # Check for mandatory parameters
        for param in config["mandatory_params"]:
            if param not in kwargs:
                raise ValueError(f"Missing parameter: {param}")

        # Collect valid parameters
        all_params = set(config["mandatory_params"])
        if "optional_params" in config:
            all_params.update(config["optional_params"])
        
        valid_params = {k: kwargs[k] for k in all_params if k in kwargs}

        # Instantiate the item
        return config[cls.ITEM_KEY](*args, **valid_params)

    @classmethod
    def create(cls, item_name, *args, **kwargs):
        return cls._validate_and_instantiate(item_name, *args, **kwargs)