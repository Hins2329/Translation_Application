package com.translation_service.translation_service;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;

@SpringBootApplication
//@EnableEurekaClient
@EnableFeignClients
public class TranslationServiceApplication {

	public static void main(String[] args) {
		SpringApplication.run(TranslationServiceApplication.class, args);
	}

}
