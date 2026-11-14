from src.nlp.normalize import normalize
from src.nlp.extract import extract_model_code
from src.nlp.edit_distance import distance, correct_products
from src.nlp.eliza import reply


def test_turkish_and_repeats():
    assert normalize('İYİ ÇOKKK') == 'iyi çokk'
    assert normalize('ISIK') == 'ısık'


def test_code():
    assert extract_model_code('broken dw-6534') == 'DW-6534'
    assert extract_model_code('hello') is None


def test_distance_and_correction():
    assert distance('kitten', 'sitting') == 3
    assert correct_products('dishwacher broken') == 'dishwasher broken'


def test_eliza():
    assert reply('merhaba')
    assert reply('merhaba cihaz bozuk') is None
