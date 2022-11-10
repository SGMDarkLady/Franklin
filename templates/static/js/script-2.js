function init(){
    const { title, character, place, keyword } = JSON.parse(localStorage.getItem("prompt_info"));
    console.log(title, character, place, keyword);

    document.getElementById("user-title").innerText=title;
    document.getElementById("character-info").innerText=character;
    document.getElementById("place-info").innerText=place;
    document.getElementById("keyword-info").innerText=keyword;
}
init();