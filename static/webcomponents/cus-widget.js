var CusWidgetProto = Object.create(HTMLDivElement.prototype);
CusWidgetProto.createdCallback = function() {
    var template = document.querySelector("link[rel='import'][href$='cus-widget.html']")
            .import.content.getElementById("cus-widget_template");
    this.createShadowRoot().appendChild(template.content.cloneNode(true));
};

CusWidget = document.register("cus-widget", {prototype: CusWidgetProto});
