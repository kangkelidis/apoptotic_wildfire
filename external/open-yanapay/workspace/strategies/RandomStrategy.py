from src.adaptation_strategy import AdaptationStrategy, Survivor


class RandomStrategy(AdaptationStrategy):
    """
    Randomly choose between asking for help and calling staff.
    """
    def get_robot_action(self,
                         simulation_id: str,
                         candidate_helper: Survivor,
                         victim: Survivor,
                         helper_victim_distance: float,
                         first_responder_victim_distance: float) -> str:
        roll = self._deterministic_roll(
            self.__class__.__name__,
            simulation_id,
            candidate_helper.gender,
            candidate_helper.cultural_cluster,
            candidate_helper.age,
            victim.gender,
            victim.cultural_cluster,
            victim.age,
            round(float(helper_victim_distance), 4),
            round(float(first_responder_victim_distance), 4),
        )
        if roll < 0.5:
            return self.ASK_FOR_HELP_ROBOT_ACTION
        else:
            return self.CALL_STAFF_ROBOT_ACTION
