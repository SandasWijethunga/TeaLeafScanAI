// TeaCare AI — client-side image preview before upload

document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file");
    const preview = document.getElementById("preview");

    if (!fileInput || !preview) return;

    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];

        if (!file) {
            preview.classList.add("d-none");
            return;
        }

        const reader = new FileReader();
        reader.onload = function (event) {
            preview.src = event.target.result;
            preview.classList.remove("d-none");
        };
        reader.readAsDataURL(file);
    });
});
