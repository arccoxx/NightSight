"""Model registry for dynamic model instantiation."""

from typing import Dict, Type, Any, Optional, Callable
from nightsight.core.base import BaseModel, BaseEnhancer


class ModelRegistry:
    """
    Registry for models and enhancers.

    Allows dynamic registration and instantiation of models by name.

    Example:
        >>> @ModelRegistry.register("my_model")
        ... class MyModel(BaseModel):
        ...     pass

        >>> model = ModelRegistry.create("my_model", **config)
    """

    _models: Dict[str, Type[BaseModel]] = {}
    _enhancers: Dict[str, Type[BaseEnhancer]] = {}
    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        default_config: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator to register a model class.

        Args:
            name: Name to register the model under
            default_config: Optional default configuration

        Returns:
            Decorator function
        """
        def decorator(model_class: Type):
            if issubclass(model_class, BaseModel):
                cls._models[name] = model_class
            elif issubclass(model_class, BaseEnhancer):
                cls._enhancers[name] = model_class
            else:
                raise TypeError(
                    f"{model_class} must be subclass of BaseModel or BaseEnhancer"
                )

            if default_config is not None:
                cls._configs[name] = default_config

            return model_class

        return decorator

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a model class directly."""
        cls._models[name] = model_class

    @classmethod
    def register_enhancer(cls, name: str, enhancer_class: Type[BaseEnhancer]) -> None:
        """Register an enhancer class directly."""
        cls._enhancers[name] = enhancer_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """
        Create a model instance by name.

        Args:
            name: Registered model name
            **kwargs: Model configuration (overrides defaults)

        Returns:
            Model instance
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")

        # Merge default config with provided kwargs
        config = cls._configs.get(name, {}).copy()
        config.update(kwargs)

        return cls._models[name](**config)

    @classmethod
    def create_enhancer(cls, name: str, **kwargs) -> BaseEnhancer:
        """Create an enhancer instance by name."""
        if name not in cls._enhancers:
            available = list(cls._enhancers.keys())
            raise ValueError(f"Unknown enhancer: {name}. Available: {available}")

        config = cls._configs.get(name, {}).copy()
        config.update(kwargs)

        return cls._enhancers[name](**config)

    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())

    @classmethod
    def list_enhancers(cls) -> list:
        """List all registered enhancers."""
        return list(cls._enhancers.keys())

    @classmethod
    def get_config(cls, name: str) -> Dict[str, Any]:
        """Get the default configuration for a model."""
        return cls._configs.get(name, {}).copy()


# Global registry instance for convenience
MODELS = ModelRegistry()
