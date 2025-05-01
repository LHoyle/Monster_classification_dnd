#!/usr/bin/env python3


################################################################
#
# These custom classes help with pipeline building and debugging
#
import sklearn.base

class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Just a placeholder with no actions on the data.
    """
    
    def __init__(self):
        return

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        return X

class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Pipeline member to display the data at this stage of the transformation.
    """
    
    def __init__(self, title):
        self.title = title
        return

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        print("{}::type(X)".format(self.title), type(X))
        print("{}::X.shape".format(self.title), X.shape)
        if not isinstance(X, pd.DataFrame):
            print("{}::X[0]".format(self.title), X[0])
        print("{}::X".format(self.title), X)
        return X

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
        # skip "name", "URL"

        self.mCategoricalPredictors = ["isNamedCreature","isNpc","hp/formula","alignment","type",
                                       "type/tags","swarm","size", "legendary",
                                       "source","initiative/advantageMode",'senseTags','speed/burrow/condition',
                                       'speed/climb/condition','speed/fly/condition','speed/swim/condition',
                                       'trait','traitTags','spellcasting','damageTagsSpell','spellcastingTags',
                                       'conditionInflictSpell','action','actionTags','bonus','reaction','conditionInflict',
                                       'damageTags','savingThrowForced','legendary','damageTagsLegendary','savingThrowForcedLegendary',
                                       'conditionInflictLegendary','mythic','resist','vulnerable','immune','conditionImmune',
                                       'languages','languageTags','items','environment','miscTags','source','familiar','summonedBySpell',
                                       'summonedByClass','variant','reprinted','sidekickHidden']
        self.mNumericalPredictors = ["ac", "hp/average", "str", "dex","con","int","wis","cha","save/cha","save/int","save/wis","save/con","save/dex","save/str","skill/arcana",
                                     "skill/history","skill/insight",'skill/perception','skill/performance','skill/stealth','skill/investigation','skill/acrobatics','skill/animal handling',
                                     'skill/athletics','skill/deception','skill/intimidation','skill/medicine','skill/nature','skill/persuasion','skill/religion','skill/sleight of hand','skill/survival', 
                                     "passive",'speed/walk','speed/burrow','speed/climb',
                                     'speed/fly','speed/swim','legendaryActions','summonedBySpellLevel']
        self.mLabels = ["cr"]
        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        
        if do_predictors:
            if do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
            
        return

    def fit( self, X, y=None ):
        # no fit necessary
        self.is_fitted_ = True
        return self

    def transform( self, X, y=None ):
        # only keep columns selected
        values = X[self.mAttributes]
        return values

#
# These custom classes help with pipeline building and debugging
#
################################################################
