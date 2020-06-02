
from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
import os
import json


#make pokedict from bulbapedia, no need to run
def make_pokedict():
    pokedex_url = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number"
    pokemon_dict = {}

    header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
    req = urllib.request.Request(pokedex_url, headers= header)
    pokedex_html = urllib.request.urlopen(req)
    pokedex_soup = BeautifulSoup(pokedex_html.read(), "html.parser")
    pokedex_int1 = pokedex_soup.findAll("tr", {"style": "background:#FFF"})

    for i in pokedex_int1:
        num = i.select("td")[1].contents[0]
        name = i.th.a["title"]
        type_ = []
        for typ in i.findAll("span"):
            type_.append(typ.contents[0])
        num = num[2:-1]
        try:
            int(num)
            pokemon_dict[num] = name, type_
        except:
            pass

    os.chdir("..\\Pokemon_pic")
    with open("pokemon_dict.json", "w") as f:
        json.dump(pokemon_dict, f, indent = 2)


#download picture of pokemon from bulpedia, take a ranged list e.g. [1,151]

def download_pic(list_range):
    pokemon_range = list_range
    pokemon_range = range(pokemon_range[0],pokemon_range[1])

    os.chdir("..\\Pokemon_pic")
    with open("pokemon_dict.json", "r") as f:
        pokemon_dict = json.load(f)
    os.chdir("..\\Code")

    selected_pokemon = {k:v for k, v in pokemon_dict.items() if int(k) in pokemon_range}

    for pokemon in selected_pokemon:
        pokemon_id = pokemon
        pokemon_name = selected_pokemon[pokemon][0]
        pokemon_name = pokemon_name.replace(" ","_")
        the_pokemon = pokemon_id + pokemon_name
        url = "https://bulbapedia.bulbagarden.net/wiki/File:" + the_pokemon + ".png"
        print(url)
        try:
            header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
            req = urllib.request.Request(url, headers=header)
            poke_html = urllib.request.urlopen(req)
            poke_soup = BeautifulSoup(poke_html.read(), "html.parser")
            poke_int1 = poke_soup.findAll("img", {"width": "600"})
            os.chdir("..\\Pokemon_pic")
            pic_url = "http:" + poke_int1[0]["src"]

            req2 = urllib.request.Request(pic_url, headers=header)
            pic = urllib.request.urlopen(req2).read()
            os.chdir("..\\Pokemon_pic")
            with open(f"{the_pokemon}.png", "wb") as f:
                f.write(pic)
        except UnicodeEncodeError:
            print(f"UnicodeError occurs when downloading {the_pokemon}")


download_pic([154,386])
