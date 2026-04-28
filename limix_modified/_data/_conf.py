#Dieser Code definiert ein Python-Wörterbuch CONF, das verschiedene Konfigurationsparameter für ein Datenverarbeitungssystem speichert
# Hier ist eine Erklärung für jedes Schlüssel-Wert-Paar im Wörterbuch:

#likelihoods: Eine Menge von Likelihoods, die für statistische Modelle verwendet werden können. Die Likelihoods sind "normal", "bernoulli", "probit", "binomial" und "poisson".
#targets: Eine Menge von Zielen oder Variablen, auf die sich die Analyse beziehen kann. Dies können beispielsweise Merkmale wie "trait" (Eigenschaft), "covariate" (Kovariate), "genotype" (Genotyp), "covariance" (Kovarianz) usw. sein.
#filetypes: Eine Menge von Dateitypen, die von dem System unterstützt werden. In diesem Fall sind es "csv" und "bed".
#dim_axis: Ein Wörterbuch, das die Dimensionen (Achsen) für verschiedene Datenobjekte definiert. Zum Beispiel wird die Dimension "sample" auf Achse 0, "trait" und "candidate" auf Achse 1 usw. abgebildet.
#dim_names: Eine Menge von Dimensionen, die im System verwendet werden. Dies sind die Namen der Dimensionen wie "sample", "candidate", "covariate" und "trait".
#data_synonym: Eine Zuordnung von Synonymen für verschiedene Datenobjekte. Zum Beispiel werden "y" und "trait" als Synonyme verwendet, ebenso wie "G" und "genotype", "M" und "covariate" usw.
#data_dims: Eine Zuordnung von Dimensionen für verschiedene Datenobjekte. Zum Beispiel hat "trait" die Dimensionen ["sample", "trait"], "genotype" hat ["sample", "candidate"] usw.
#varname_to_target: Eine Zuordnung von Variablennamen zu Zielen. Zum Beispiel wird "y" dem Ziel "trait" zugeordnet, "M" dem Ziel "covariate" usw.
#target_to_varname: Eine Zuordnung von Zielen zu Variablennamen. Das ist das Gegenstück zu varname_to_target.

CONF = {
    "likelihoods": set(["normal", "bernoulli", "probit", "binomial", "poisson"]),
    "targets": set(
        [
            "trait",
            "covariate",
            "covariance",
            "genotype",
            "covariate",
            "inter0",
            "inter1",
            "env",
            "env0",
            "env1",
        ]
    ),
    "filetypes": set(["csv", "bed"]),
    "dim_axis": {
        "sample": 0,
        "trait": 1,
        "candidate": 1,
        "covariate": 1,
        "sample_0": 0,
        "sample_1": 1,
    },
    "dim_names": {"sample", "candidate", "covariate", "trait"},
    "data_synonym": {
        "y": "trait",
        "trait": "y",
        "G": "genotype",
        "genotype": "G",
        "M": "covariate",
        "covariate": "M",
        "K": "covariance",
        "covariance": "K",
    },
    "data_dims": {
        "trait": ["sample", "trait"],
        "genotype": ["sample", "candidate"],
        "covariate": ["sample", "covariate"],
        "covariance": ["sample_0", "sample_1"],
        "inter0": ["sample", "inter"],
        "inter1": ["sample", "inter"],
        "env": ["sample", "env"],
        "env0": ["sample", "env"],
        "env1": ["sample", "env"],
    },
    "varname_to_target": {
        "y": "trait",
        "M": "covariate",
        "G": "genotype",
        "K": "covariance",
    },
    "target_to_varname": {
        "trait": "y",
        "covariate": "M",
        "genotype": "G",
        "covariance": "K",
    },
}
