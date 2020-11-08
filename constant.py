# -*- coding: utf-8 -*-

DATASETS = {"ctb", "ptb", "spmrl"}

SPMRL_SPLITS = ['dev', 'test', 'train', 'train5k']
SPMRL_LANGS = ['BASQUE', 'GERMAN', 'FRENCH', 'HEBREW', 'HUNGARIAN', 'KOREAN', 'POLISH', 'SWEDISH'] 
SPMRL_ROOTS = ['TOP', 'VROOT', '', 'TOP', 'ROOT', 'TOP', None, ''] 

PTB_TRAIN_SEC = [
    '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
    '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
]
PTB_TEST_SEC = ['23'] 
PTB_DEV_SEC = ['22']

STRIPPED_TAGS = {

"ENGLISH" : 
    [
        "-NONE-", # :
        ",", # :
        ".", # :
        "$", # :
        "``", # :
        "''", # :
        ":", # :
        "-RRB-", # :	
        "-LRB-", # :
        "#", #:
    ],
"CHINESE" : 
    [
        "-NONE-", # : empty
        "PU", # : punctuation
    ],
"BASQUE" :
    [
        "PUNT", # : punctuation
        "BEREIZ", # : " or ( or )
    ],
"GERMAN" : 
    [
        "$.", # : 
        "$,", # : 
        "$LRB", # :
        #"XY", # : * or _ 
    ],
    #CARD : number
"FRENCH" : 
    [
        "7", # :
        "PONCT", # : 
    ],
"HEBREW" : 
    [
        "yyELPS", # :
        "yyEXCL", # :
        "yyQM", # :
        "yySCLN", # :
        "yyCLN", # :
        "yyLRB", # :
        "yyRRB", # :
        "yyDASH", # :
        "yyQUOT", # :
        "yyDOT", # :
        "yyCM", # :
    ], 
    #NCD : number
"HUNGARIAN" :
    [
        "PUNC", # :
        "K", # : ellipse
    ],
"KOREAN" :
    [
        "sl", # : 
        "sr", # :
        "sp", # :
        "sf", #:
    ], 
    #su: %
"POLISH" :
    [
        "interp", # : 
    ],
"SWEDISH" :
    [
        "MAD", # :
        "MID", # :
        "PAD", # :
    ],
    #RG : number
}
