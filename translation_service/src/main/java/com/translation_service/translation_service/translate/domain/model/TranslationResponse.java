package com.translation_service.translation_service.translate.domain.model;

public class TranslationResponse {
    private String translation;

    public TranslationResponse() {
    }

    public TranslationResponse(String translation) {
        this.translation = translation;
    }

    public String getTranslation() {
        return translation;
    }

    public void setTranslation(String translation) {
        this.translation = translation;
    }
}
