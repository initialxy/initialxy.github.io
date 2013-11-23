// Import style sheet. I shouldn't have to do this!
var styleSheet = document.currentScript.ownerDocument
        .querySelector("link[rel='stylesheet'][href$='cus-widget.css']");
document.head.appendChild(styleSheet.cloneNode(true));

// Create custom widget.
var CusWidgetProto = Object.create(HTMLDivElement.prototype);
CusWidgetProto.createdCallback = function() {
    var template = document.querySelector("link[rel='import'][href$='cus-widget.html']")
            .import.getElementById("cus-widget_template");
    this.createShadowRoot().appendChild(template.content.cloneNode(true));
};

// Try catch is here is here because of a polymer bug.
// https://github.com/Polymer/polymer/issues/290
try {
    CusWidget = document.register("cus-widget", {prototype: CusWidgetProto});
} catch (e) {
}
