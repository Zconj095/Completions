from sklearn.neural_network import MLPClassifier
import numpy as np

class Skill:
    def __init__(self, base_effect):
        self.level = 1
        self.exp = 0
        self.base_effect = base_effect
        self.modifiers = []
        
    def acquire(self, points, prereqs):
        # Check prereqs and points
        return True 
    
    def progress(self, exp):
        self.exp += exp
        if self.exp >= 100*self.level:
            self.level += 1
            
    def get_effect(self):
        effect = self.base_effect*self.level
        for modifier in self.modifiers:
            effect *= modifier
        return effect
        
class SkillTree:
    def __init__(self):
        self.skills = []
        self.paths = [[] for i in range(3)]
        
    def add_skill(self, skill):
        self.skills.append(skill)
        
    def choose_path(self, path, strategy):
        # Skill choice logic
        return self.paths[path]
    
    def get_usage(self):
        usage = []
        for skill in self.skills:
            usage.append(skill.level*skill.get_effect())
        return usage
    
    def evolve(self, usages, fitness_fn):
        model = MLPClassifier(hidden_layer_sizes=(64,32))
        model.fit(usages, fitness_fn(usages))
        return model

def adjust_balance(balances, feedback):
    return balances + np.array(feedback)/100