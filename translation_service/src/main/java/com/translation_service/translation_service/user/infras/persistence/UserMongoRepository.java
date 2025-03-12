package com.translation_service.translation_service.user.infras.persistence;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.data.mongodb.repository.Update;

import java.util.Optional;

public interface UserMongoRepository extends MongoRepository<UserDocument, Integer> {

    @Query("{ '_id': ?0 }")
    @Update("{ '$push': { 'sentences': ?1 } }")
    void addSentence(String id, String sentence);

    @Query("{ '_id': ?0 }")
    @Update("{ '$setOnInsert': ?1 }")
    void createUser(String id, UserDocument user);

}
