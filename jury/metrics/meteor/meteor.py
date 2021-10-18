from jury.metrics._core import MetricAlias
from jury.metrics.meteor.meteor_for_language_generation import MeteorForLanguageGeneration

__main_class__ = "Meteor"


class Meteor(MetricAlias):
    _SUBCLASS = MeteorForLanguageGeneration
