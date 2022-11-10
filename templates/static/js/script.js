var localStorageEnabled = false;
try { localStorageEnabled = !!localStorage; } catch(e) {};

if (localStorageEnabled) localStorage.clear();

function next()  {
    window.location.href = 'generate-2.html';
}

function promptText(){
    const title = document.getElementById('fairytale-title').value;
    const character_name = document.getElementById('fairytale-character').value;
    const place = document.getElementById('fairytale-place').value;
    const keyword_1 = document.getElementById('keyword_1').value;
    const keyword_2 = document.getElementById('keyword_2').value;
    const keyword_3 = document.getElementById('keyword_3').value;
    const keywords = `${keyword_1}, ${keyword_2}, ${keyword_3}`
    var prompt = {
        "title": title,
        "character":character_name,
        "place":place,
        "keyword": keywords
    };

    localStorage.setItem("prompt_info", JSON.stringify(prompt));
}

function inputPrompt(){
    const input = `제목: ${title} / 키워드: ${character_name}, ${place}, ${keyword_1}, ${keyword_2}, ${keyword_3}`
    return input;
}