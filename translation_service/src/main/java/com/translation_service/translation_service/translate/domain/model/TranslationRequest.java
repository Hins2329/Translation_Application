package com.translation_service.translation_service.translate.domain.model;

public class TranslationRequest {
    private String sentence;

    public TranslationRequest() {
    }

    public TranslationRequest(String sentence) {
        this.sentence = sentence;
    }


    public String getSentence() {
        return sentence;
    }

    public void setSentence(String sentence) {
        this.sentence = sentence;
    }

    @Override
    public String toString() {
        return "TranslationRequest{" +
                "sentence='" + sentence + '\'' +
                '}';
    }
}
